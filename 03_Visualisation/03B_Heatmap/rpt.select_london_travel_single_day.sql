USE [test1]
GO
/****** Object:  StoredProcedure [rpt].[select_london_travel_single_day]    Script Date: 10.11.2021 23:55:42 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
-- =============================================
-- Author:		Nicolas Kepper
-- Create date: 09.11.2021
-- Description:	<Description,,>
-- =============================================
ALTER PROCEDURE [rpt].[select_london_travel_single_day] 
	@datumid int
AS
BEGIN
	-- SET NOCOUNT ON added to prevent extra result sets from
	-- interfering with SELECT statements.
	SET NOCOUNT ON;

	SELECT DatumID, b.name, b.gss_code, --v.grocery_pharmacy as parks
		--v.parks, 
		--v.residential as parks, 
		--v.retail_recreation as parks--, 
		--v.transit as parks,
		 v.workplaces as parks
	FROM dbo.fact_view v
	INNER JOIN dbo.Borough b ON v.BoroughID = b.ID
	WHERE DatumID = @datumid
END